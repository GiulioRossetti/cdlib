

def write_community_csv(communities,  path, delimiter=",", community_id=False):
    """

    :param communities:
    :param path:
    :param delimiter:
    :param community_id:
    :return:
    """
    with open(path, "w") as f:
        for cid, community in enumerate(communities):
            res = delimiter.join(list(map(str, community)))
            if community_id:
                res = "%s\t%s\n" % (cid, res)
            f.write("%s\n" % res)


def read_community_csv(path, delimiter=",", community_id=False, nodetype=str):
    """

    :param path:
    :param delimiter:
    :param community_id:
    :param nodetype:
    :return:
    """
    communities = []
    with open(path) as f:
        for row in f:
            if community_id:
                cid, row = row.split("\t")
            community = list(map(nodetype, row.rstrip().split(delimiter)))
            communities.append(tuple(community))

    return communities
