__all__ = [
    "backward_event_names",
    "forward_event_names",
    "colormap",
    "get_group_attribute_values",
]


def backward_event_names() -> list:
    """
    return the list of backward event names
    """
    return [
        "Birth",
        "Accumulation",
        "Growth",
        "Expansion",
        "Continuation",
        "Merge",
        "Offspring",
        "Reorganization",
    ]


def forward_event_names() -> list:
    """
    return the list of forward event names
    """
    return [
        "Death",
        "Dispersion",
        "Shrink",
        "Reduction",
        "Continuation",
        "Split",
        "Ancestor",
        "Disassemble",
    ]


def colormap() -> dict:
    """
    return a dictionary of colors for each event type.
    this is used to color the events in the visualization
    """

    return {
        "Birth": " #808000",
        "Accumulation": "#4CC89F",
        "Growth": "#929292",
        "Expansion": "#5C5C5C",
        "Continuation": "#CFBAE1",
        "Merge": "#E34856",
        "Offspring": "#0DAAE9",
        "Reorganization": "#FFA500",
        "Death": " #808000",
        "Dispersion": "#4CC89F",
        "Shrink": "#929292",
        "Reduction": "#5C5C5C",
        "Split": "#E34856",
        "Ancestor": "#0DAAE9",
        "Disassemble": "#FFA500",
    }


def get_group_attribute_values(lc: object, target: str, attr_name: str) -> list:
    """
    retrieve the list of attributes of the elements in a set

    :param lc: a LifeCycle object
    :param target: the id of the set
    :param attr_name: the name of the attribute
    :return: a list of attributes corresponding to the elements in the set
    """

    tid = int(target.split("_")[0])
    attributes = list()

    for elem in lc.get_group(target):
        attributes.append(lc.get_attributes(attr_name, of=elem)[tid])
    return attributes
