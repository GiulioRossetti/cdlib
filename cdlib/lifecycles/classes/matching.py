import json
from collections import defaultdict

__all__ = ["CommunityMatching"]


class CommunityMatching(object):
    """
    A class to represent and analyze temporally-evolving groups.
    """

    def __init__(self, dtype: type = int) -> None:

        self.dtype = dtype
        self.tids = []
        self.named_sets = defaultdict(set)
        self.tid_to_named_sets = defaultdict(list)
        self.attributes = defaultdict(dict)

    ############################## Convenience get methods ##########################################
    def temporal_ids(self) -> list:
        """
        retrieve the temporal ids of the CommunityMatching.
        Temporal ids are integers that represent the observation time of a partition.

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([{"a", "b"}, {"c", "d"}]) # at time 0
        >>> lc.__add_partition([{"a", "b"}, {"c"}]) # at time 1
        >>> lc.temporal_ids()
        [0, 1]
        """
        return self.tids

    def slice(self, start: int, end: int) -> object:
        """
        slice the CommunityMatching to keep only a given interval

        :param start: the start of the interval
        :param end: the end of the interval
        :return: a new CommunityMatching object

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([{5,7}, {6,8}])
        >>> lc.__add_partition([{5,7}, {1,6,8}])
        >>> sliced = lc.slice(0, 1)

        """
        temp = CommunityMatching(self.dtype)
        temp.tids = self.tids[start:end]
        temp.named_sets = {
            k: v
            for k, v in self.named_sets.items()
            if int(k.split("_")[0]) in temp.tids
        }
        temp.tid_to_named_sets = {
            k: v for k, v in self.tid_to_named_sets.items() if int(k) in temp.tids
        }
        temp_attrs = {}
        for attr_name, attr in self.attributes.items():
            temp_attrs[attr_name] = {k: v for k, v in attr.items() if k in temp.tids}
        temp.attributes = temp_attrs
        return temp

    def universe_set(self) -> set:
        """
        retrieve the universe set.
        The universe set is the union of all sets in the CommunityMatching

        :return: the universe set

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]]) # at time 0
        >>> lc.__add_partition([{5,7}, {6,8}]) # at time 1
        >>> lc.universe_set()
        {1, 2, 3, 4, 5, 6, 7, 8}
        """
        universe = set()
        for set_ in self.named_sets.values():
            universe = universe.union(set_)
        return universe

    def groups_ids(self) -> list:
        """
        retrieve the group ids of the CommunityMatching. Each id is of the form 'tid_gid' where tid is the temporal id and
        gid is the group id. The group id is a unique identifier of the group within the temporal id.

        :return: a list of ids of the temporal groups

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([{5,7}, {6,8}])
        >>> lc.groups_ids()
        ['0_0', '0_1', '1_0', '1_1']
        """
        return list(self.named_sets.keys())

    ############################## Partition methods ##########################################
    def __add_partition(self, partition: list) -> None:
        """
        add a partition to the CommunityMatching. A partition is a list of sets observed at a given time instant. Each
        partition will be assigned a unique id (tid) corresponding to the observation time, and each set in the
        partition will be assigned a unique name

        :param partition: a collection of sets
        :return: None

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([{5,7}, {6,8}])
        """

        tid = len(self.tids)
        self.tids.append(tid)

        for i, group in enumerate(partition):
            name = str(tid) + "_" + str(i)
            self.tid_to_named_sets[str(tid)].append(name)

            if self.dtype in [int, float, str]:
                try:
                    self.named_sets[name] = set(group)
                except TypeError:  # group is not iterable (only 1 elem)
                    tmp = set()
                    tmp.add(group)
                    self.named_sets[name] = tmp

            elif self.dtype == dict:
                for elem in group:
                    to_str = json.dumps(elem)
                    self.named_sets[name].add(to_str)

            elif self.dtype == list:
                for elem in group:
                    to_str = str(elem)
                    self.named_sets[name].add(to_str)
            else:
                raise NotImplementedError("dtype not supported")

    def set_temporal_clustering(self, partitions: object) -> None:
        """
        add multiple partitions to the CommunityMatching.

        :param partitions: a list of partitions
        :return: None

        :Example:
        >>> lc = CommunityMatching()
        >>> partitions = [
        >>>     [[1,2], [3,4,5]],
        >>>     [{5,7}, {6,8}]
        >>> ]
        >>> lc.set_temporal_clustering(partitions)
        """
        tids = partitions.get_observation_ids()
        for t in tids:
            self.__add_partition(partitions.get_clustering_at(t).communities)

    def get_partition_at(self, tid: int) -> list:
        """
        retrieve a partition by id

        :param tid: the id of the partition to retrieve
        :return: the partition corresponding to the given id

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([{5,7}, {6,8}, {9}])
        >>> lc.get_partition_at(0)
        ['0_0', '0_1']
        >>> lc.get_partition_at(1)
        ['1_0', '1_1', '1_2']
        """
        if str(tid) not in self.tid_to_named_sets:
            return []
        return self.tid_to_named_sets[str(tid)]

    ############################## Attribute methods ##########################################
    def set_attributes(self, attributes: dict, attr_name: str) -> None:
        """
        set the temporal attributes of the elements in the CommunityMatching

        The temporal attributes must be provided as a dictionary keyed by the element id and valued by a dictionary
        keyed by the temporal id and valued by the attribute value.

        :param attr_name: the name of the attribute
        :param attributes: a dictionary of temporal attributes
        :return: None

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> attributes = {
        >>>     1: {0: 'red', 1: 'blue'}, # element 1 is red at time 0 and blue at time 1
        >>>     2: {0: 'green', 1: 'magenta'} # element 2 is green at time 0 and magenta at time 1
        >>> }
        >>> lc.set_attributes(attributes, attr_name="color")
        """
        self.attributes[attr_name] = attributes

    def get_attributes(self, attr_name, of=None) -> dict:
        """
        retrieve the temporal attributes of the CommunityMatching

        :param attr_name: the name of the attribute
        :param of: the element for which to retrieve the attributes. If None, all attributes are returned

        :return: a dictionary keyed by element id and valued by a dictionary keyed by temporal id and valued by the attribute value


        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> attributes = {
        >>>     1: {0: 'red', 1: 'blue'}, # element 1 is red at time 0 and blue at time 1
        >>>     2: {0: 'green', 1: 'magenta'} # element 2 is green at time 0 and magenta at time 1
        >>> }
        >>> lc.set_attributes(attributes, attr_name="color")
        >>> lc.get_attributes("color")
        {1: {0: 'red', 1: 'blue'}, 2: {0: 'green', 1: 'magenta'}}
        >>> lc.get_attributes("color", of=1) # get the attributes of element 1
        {0: 'red', 1: 'blue'}
        """
        if of is None:
            return self.attributes[attr_name]
        else:
            return self.attributes[attr_name][of]

    ############################## Set methods ##########################################
    def get_group(self, gid: str) -> set:
        """
        retrieve a group by id

        :param gid: the name of the group to retrieve
        :return: the group corresponding to the given name

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.get_group("0_0")
        {1, 2}
        """
        return self.named_sets[gid]

    def group_iterator(self, tid: int = None) -> iter:
        """
        returns an iterator over the groups of the CommunityMatching.
        if a temporal id is provided, it will iterate over the groups observed at that time instant

        :param tid: the temporal id of the groups to iterate over. Default is None
        :return: an iterator over the groups

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> for set_ in lc.group_iterator():
        >>>     print(set_)
        {1, 2}
        {3, 4, 5}
        {1, 2, 3}
        {4, 5}
        """
        if tid is None:
            yield from self.named_sets.values()
        else:
            for name in self.get_partition_at(tid):
                yield self.named_sets[name]

    def filter_on_group_size(self, min_size: int = 1, max_size: int = None) -> None:
        """
        remove groups that do not meet the size criteria

        :param min_size: the minimum size of the groups to keep
        :param max_size: the maximum size of the groups to keep
        :return: None

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> lc.filter_on_group_size(min_size=3) # remove groups with less than 3 elements
        >>> lc.groups_ids() # only groups 1_0 and 1_1 remain
        ['0_1', '1_0']
        """

        if max_size is None:
            max_size = len(self.universe_set())

        for name, set_ in self.named_sets.copy().items():
            if len(set_) < min_size or len(set_) > max_size:
                del self.named_sets[name]
                self.tid_to_named_sets[name.split("_")[0]].remove(name)

    ############################## Element-centric methods ##########################################
    def get_element_membership(self, element: object) -> list:
        """
        retrieve the list of sets that contain a given element

        :param element: the element for which to retrieve the memberships
        :return: a list of set names that contain the given element

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> lc.get_element_membership(1)
        ['0_0', '1_0']
        """

        memberships = list()
        for name, set_ in self.named_sets.items():
            if element in set_:
                memberships.append(name)
        return memberships

    def get_all_element_memberships(self) -> dict:
        """
        retrieve the list of sets that contain each element in the CommunityMatching

        :return: a dictionary keyed by element and valued by a list of set names that contain the element

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> lc.get_all_element_memberships()
        """

        memberships = defaultdict(list)

        for element in self.universe_set():
            for name, set_ in self.named_sets.items():
                if element in set_:
                    memberships[element].append(name)

        return memberships

    ############################## Flow methods ##########################################
    def group_flow(self, target: str, direction: str, min_branch_size: int = 1) -> dict:
        """
        compute the flow of a group w.r.t. a given temporal direction. The flow of a group is the collection of groups that
        contain at least one element of the target group, Returns a dictionary keyed by group name and valued by the
        intersection of the target group and the group corresponding to the key.

        :param target: the name of the group to analyze
        :param direction: the temporal direction in which the group is to be analyzed
        :param min_branch_size: the minimum size of the intersection between the target group and the group corresponding
        :return: a dictionary keyed by group name and valued by the intersection of the target group and the group

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> lc.group_flow("0_0", "+")
        {'1_0': {1, 2}}
        """
        flow = dict()
        tid = int(target.split("_")[0])
        if direction == "+":
            ref_tid = tid + 1
        elif direction == "-":
            ref_tid = tid - 1
        else:
            raise ValueError("direction must either be + or -")
        reference = self.get_partition_at(ref_tid)
        target_set = self.get_group(target)

        for name in reference:
            set_ = self.get_group(name)
            branch = target_set.intersection(set_)
            if len(branch) >= min_branch_size:
                flow[name] = branch
        return flow

    def all_flows(self, direction: str, min_branch_size: int = 1) -> dict:
        """
        compute the flow of all groups w.r.t. a given temporal direction

        :param direction: the temporal direction in which the sets are to be analyzed
        :param min_branch_size: the minimum size of a branch to be considered
        :return: a dictionary keyed by group name and valued by the flow of the group

        :Example:
        >>> lc = CommunityMatching()
        >>> lc.__add_partition([[1,2], [3,4,5]])
        >>> lc.__add_partition([[1,2,3], [4,5]])
        >>> lc.all_flows("+")
        {'0_0': {'1_0': {1, 2}}, '0_1': {'1_0': {3}, '1_1': {4, 5}}}

        """
        all_flows = dict()
        for name in self.named_sets:
            all_flows[name] = self.group_flow(
                name, direction, min_branch_size=min_branch_size
            )

        return all_flows
