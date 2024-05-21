from cdlib.lifecycles.classes.matching import CommunityMatching
from cdlib.lifecycles.algorithms.measures import *
from cdlib.lifecycles.utils import *

__all__ = [
    "analyze_all_flows",
    "analyze_flow",
    "events_all",
    "facets",
    "event_weights",
    "event",
    "event_weights_from_flow",
]


def _analyze_one_struct(target, reference) -> dict:
    # nb reference sets here are already filtered by minimum branch size

    ids_for_entropy = []
    # els_in_branches = set()
    for i, r in enumerate(reference):
        branch = target.intersection(r)
        ids_for_entropy.extend([str(i)] * len(branch))
        # els_in_branches.update(branch)
    # newels_ids = [str(j+len(reference)) for j in range(len(target.difference(els_in_branches)))]
    # ids_for_entropy.extend(newels_ids)

    return {
        "Unicity": facet_unicity(ids_for_entropy),
        "Identity": facet_identity(target, reference),
        "Outflow": facet_outflow(target, reference),
        "size": len(target),
    }


def _analyze_one_attr(target, reference, attr) -> dict:
    mca, pur = purity(target)
    try:
        ent = _normalized_shannon_entropy(target, base=2)
    except ZeroDivisionError:
        ent = 0

    return {
        f"{attr}_H": ent,
        f"{attr}_H_change": facet_metadata(target, reference, base=2),
        f"{attr}_purity": pur,
        f"{attr}_mca": mca,
    }


def event_weights_from_flow(analyzed_flows: dict, direction: str) -> dict:
    """
    Compute the event weights of the analyzed flows.

    :param analyzed_flows:  the result of the analysis of a flow
    :param direction:  the temporal direction in which the flow was analyzed
    :return: a dictionary containing the event weights
    """
    if direction not in ["+", "-"]:
        raise ValueError(f"direction must be either '+' or '-'")
    res = {}
    names = backward_event_names() if direction == "-" else forward_event_names()
    for id_, analyzed_flow in analyzed_flows.items():
        scores = _compute_event_scores(analyzed_flow)
        res[id_] = dict(zip(names, scores))

    return res


def _compute_event_scores(analyzed_flow: dict) -> list:
    return [
        (analyzed_flow["Unicity"])
        * (1 - analyzed_flow["Identity"])
        * analyzed_flow["Outflow"],
        (1 - analyzed_flow["Unicity"])
        * (1 - analyzed_flow["Identity"])
        * analyzed_flow["Outflow"],
        (analyzed_flow["Unicity"])
        * analyzed_flow["Identity"]
        * analyzed_flow["Outflow"],
        (1 - analyzed_flow["Unicity"])
        * analyzed_flow["Identity"]
        * analyzed_flow["Outflow"],
        (analyzed_flow["Unicity"])
        * analyzed_flow["Identity"]
        * (1 - analyzed_flow["Outflow"]),
        (1 - analyzed_flow["Unicity"])
        * analyzed_flow["Identity"]
        * (1 - analyzed_flow["Outflow"]),
        (analyzed_flow["Unicity"])
        * (1 - analyzed_flow["Identity"])
        * (1 - analyzed_flow["Outflow"]),
        (1 - analyzed_flow["Unicity"])
        * (1 - analyzed_flow["Identity"])
        * (1 - analyzed_flow["Outflow"]),
    ]


def events_all(lc: CommunityMatching, direction=None) -> dict:
    """
    Compute all events for a lifecycle object.

    :param lc: a LifeCycle object
    :param direction: the temporal direction in which the events are to be computed

    :return: a dictionary containing the events

    """
    if direction is None:
        direction = ["+", "-"]
    res = {}
    for d in direction:
        analyzed_flows = analyze_all_flows(lc, d)
        res[d] = event_weights_from_flow(analyzed_flows, d)
    return res


def analyze_all_flows(
    lc: CommunityMatching, direction: str, min_branch_size: int = 1, attr=None
) -> dict:
    """
    Analyze the flow of all sets in a LifeCycle object w.r.t. a given temporal direction.
    See analyze_flow for more details
    :param lc: a LifeCycle object
    :param direction: the temporal direction in which the sets are to be analyzed
    :param min_branch_size: the minimum number of elements that a branch must contain to be considered
    :param attr: the name or list of names of the attribute(s) to analyze. If None, no attribute is analyzed
    :return:
    """
    last_id = lc.temporal_ids()[-1] if direction == "+" else lc.temporal_ids()[0]
    return {
        name: analyze_flow(
            lc, name, direction, min_branch_size=min_branch_size, attr=attr
        )
        for name in lc.named_sets
        if not name.split("_")[0] == str(last_id)
    }


def analyze_flow(
    lc: CommunityMatching,
    target: str,
    direction: str,
    min_branch_size=1,
    attr: str = None,
) -> dict:
    """
    Analyze the flow of a set with respect to a given temporal direction.
    Specifically, compute the entropy of the flow, the contribution factor, the difference factor and the set size.
    If one of more attributes are specified via the attr parameter, also compute the entropy of the attribute values,
    the entropy change, the purity and the most common attribute value.
    In case min_branch_size is specified, all branches of the flow that include less than min_branch_size elements are
    discarded.
    :param lc:  a LifeCycle object
    :param target:  the name of the set to analyze
    :param direction:  the temporal direction in which the set is to be analyzed
    :param min_branch_size:  the minimum number of elements that a branch must contain to be considered
    :param attr:  the name or list of names of the attribute(s) to analyze. If None, no attribute is analyzed
    :return: a dictionary containing the analysis results
    """

    flow = lc.group_flow(target, direction=direction, min_branch_size=min_branch_size)

    reference_sets = [lc.get_group(name) for name in flow]
    analysis = _analyze_one_struct(lc.get_group(target), reference_sets)

    if attr is not None:
        attrs_to_analyze = [attr] if isinstance(attr, str) else attr
        for a in attrs_to_analyze:
            target_attrs = get_group_attribute_values(lc, target, a)
            reference_attrs = [get_group_attribute_values(lc, name, a) for name in flow]
            analysis.update(_analyze_one_attr(target_attrs, reference_attrs, a))
    return analysis


def facets(lc: CommunityMatching, target: str, direction: str) -> dict:
    """
    Compute the unicity, identity, and outflow facets of a target set in a lifecycle object.
    Also compute the size of the target set.

    :param lc: a LifeCycle object
    :param target: the name of the target set
    :param direction: the temporal direction in which the flow is to be analyzed
    :return: a dictionary containing the facets
    """
    flow = lc.group_flow(target, direction=direction)

    reference_sets = [lc.get_group(name) for name in flow]
    facets_ = _analyze_one_struct(lc.get_group(target), reference_sets)
    return facets_


def event_weights(lc: CommunityMatching, target: str, direction: str) -> dict:
    """
    Compute the event weights of a target set in a lifecycle object.

    :param lc: a LifeCycle object
    :param target: the name of the target set
    :param direction: the temporal direction in which the flow is to be analyzed
    :return: a dictionary containing the event weights
    """
    names = backward_event_names() if direction == "-" else forward_event_names()
    fscores = facets(lc, target, direction)
    res = _compute_event_scores(fscores)
    return dict(zip(names, res))


def event(lc, target, direction=None):
    """
    Compute the event type and typicality of a target set in a lifecycle.

    :param lc: lifecycle object
    :param target: name of the target set
    :param direction: temporal direction in which the flow is to be analyzed
    :return: a dictionary containing the event type and scores
    """
    if direction is None:
        direction = ["+", "-"]
    back = {}
    forward = {}
    if "-" in direction:
        back = event_typicality(event_weights(lc, target, "-"))
    if "+" in direction:
        forward = event_typicality(event_weights(lc, target, "+"))
    return {"+": forward, "-": back}
