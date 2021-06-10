# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:14:20 2020
@author: saipr
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx import community as c
import community as com
import itertools
import random
from random import sample
from networkx import algorithms as al


# --------------------Methods in Core Expansion--------------------------------#

# Managing method to call everything as need be
def findCommunities(g, tol=0.0005):
    # assign all the edges their neighborhoodOverlap
    nx.set_edge_attributes(g, computeNeighborOverlap(g), "neighborOverlap")
    # assign all the nodes their sumOverlap attributes
    nx.set_node_attributes(g, computeSumOverlap(g), "sumOverlap")
    # assign every node a coreVal of -1
    setup = list()
    setup.append(-1)
    nx.set_node_attributes(g, setup, "coreValue")

    # find all the local maximums in sumOverlap
    communities = findCores(g)
    # Assign initial cores their unique core value
    for xx in range(0, len(communities)):
        for co in communities[xx]:
            g.nodes[co]["coreValue"] = [xx]

            # sweep 1
    iterr = 0
    while True:
        changed = False
        nodeIter = iter(g.nodes)
        coreValDict = nx.get_node_attributes(g, "coreValue")
        # now go through all the nodes and edit the unassigned
        for i in range(0, len(g.nodes)):
            check = next(nodeIter)
            if coreValDict[check][0] == -1:  # if the node is unassigned
                test = findClosestCore(communities, check, g)
                if test[0] != -1 and (len(test) < (tol * len(communities))):
                    coreValDict.update({check: test})
                    changed = True
        iterr += 1

        # if something has changed, update the list
        if changed:
            nx.set_node_attributes(g, coreValDict, "coreValue")  # update core values
            # the core list is updated, let's update the community list
            nodeIter = iter(g.nodes)  # for each node in the community
            for j in range(0, len(g.nodes)):
                thisOne = next(nodeIter)  # what node we're checking
                temppp = g.nodes[thisOne]["coreValue"]
                for thisCore in temppp:
                    # check if its coreval is not negative one
                    if thisCore != -1:
                        # if it's not negative, check if it's not already in that community
                        if thisOne in communities[thisCore]:
                            pass
                        else:  # and if not, add it in
                            communities[thisCore].append(thisOne)
        else:  # if nothing's changed, break the loop
            break

    # now do the second sweep
    nodeIter = iter(g.nodes)
    coreValDict = nx.get_node_attributes(g, "coreValue")

    for i in range(0, len(g.nodes)):
        straggler = next(nodeIter)
        coreValsTemp = g.nodes[straggler]["coreValue"]
        if coreValsTemp[0] == -1:  # if the node is unassigned
            byConnections = secondSort(
                g, straggler, communities
            )  # check if it belongs to any core by # of connections
            coreValDict.update({straggler: byConnections})
            for b in byConnections:
                if b != -1:
                    communities[b].append(straggler)

    nx.set_node_attributes(g, coreValDict, "coreValue")

    # step 6: print out any unassigned nodes, and what proportion they are out of all of them
    areUnassigned(g)

    return communities


# compute neighborhood overlaps- takes in a graph, returns a dictionary to set the values of g.edges with
def computeNeighborOverlap(g):
    # creating a dictionary of edges and thier neighborhood overlap value
    edgeOverlaps = {}
    # to iterate over the edges in g
    edgeIter = iter(g.edges)  # next() returns a touple with to and from
    for i in range(0, len(g.edges)):
        thisEdge = next(edgeIter)
        overVal = -1  # current overlap value
        to = thisEdge[0]  # first element of the edge touple, to
        fromm = thisEdge[1]  # second element of the edge touple, from

        # calculate the edgeOverlap between the two and from of thisEdge
        # cuv / (ku + kv - 2 - cuv)
        cuv = sum(
            1 for e in nx.common_neighbors(g, to, fromm)
        )  # shared neighbors of the endpoints
        ku = sum(1 for e in nx.all_neighbors(g, to))
        kv = sum(1 for e in nx.all_neighbors(g, fromm))
        overVal = 0
        try:
            overVal = cuv / (ku + kv - 2 - cuv)
        except:
            pass
        edgeOverlaps.update({thisEdge: overVal})
    return edgeOverlaps


# compute a node's sumOverlap - takes a graph, returns a directory to set the values of g.nodes with
def computeSumOverlap(g):
    nodeIter = iter(g.nodes)  # our darling node iteraor
    sumOverlaps = {}

    # for each node in G
    for i in range(0, len(g.nodes)):
        thisNode = next(nodeIter)  # the current node we're focusing on
        summ = 0  # curent sum neighbor overlap
        thisNodesEdges = g.edges(thisNode)  # we need the edges of this node
        thisEdgeIter = iter(thisNodesEdges)  # and to make an iterator out of them
        # add the sum of its edges' neighbor overlaps
        try:
            while True:
                thisEdge = next(thisEdgeIter)
                summ += g.edges[thisEdge]["neighborOverlap"]
        except:
            pass
        sumOverlaps.update({thisNode: summ})
    return sumOverlaps


# find the cores of a graph - nodes with a local maximum among their peers in sumOverlap
# also assigns the cores their core index, changing it away from -1 (so unassigned are nodes with core value -1)
def findCores(g):
    # given a graph g
    # create a list 'cores' to return
    cores = list()

    # for each node in graph g
    nodeIter = iter(g.nodes)
    for i in range(0, len(g.nodes)):
        passs = False  # should we skip this node
        thisNode = next(nodeIter)
        # get all their neighbors and how many there are
        numNeigh = sum(1 for i in nx.neighbors(g, thisNode))
        neighIter = nx.neighbors(g, thisNode)

        # check if any of their neighbors are already in the core list (if so, this node is NOT the local maximum)
        for j in range(0, numNeigh):
            m = next(neighIter)
            for c in cores:
                for q in range(0, len(c)):
                    if m == c[q]:
                        passs = True

                    # else, check if this node is the local maximum
        if not passs:
            # turn the sumOverlap of all their neighbors into a list
            check = g.nodes[thisNode]["sumOverlap"]  # what we're checking for
            neighIter = nx.neighbors(
                g, thisNode
            )  # reset the neighbor iterator to the top
            againstThese = {}
            for z in range(0, numNeigh):
                thisOne = next(neighIter)
                againstThese.update({thisOne: g.nodes[thisOne]["sumOverlap"]})

            # check if their sumOverlap is higher than all of their neighbors' (np.all())
            if check > np.all(np.array(list(againstThese.values()))):
                tem = list()
                tem.append(thisNode)
                cores.append(tem)
            elif check >= np.all(
                np.array(list(againstThese.values()))
            ):  # there's a speed tie
                ties = list()
                ties.append(thisNode)
                for key, value in againstThese.items():
                    if (value == check) and (key != thisNode):
                        ties.append(key)
                cores.append(ties)  # returns a group of cores that compose the same mx
    return cores


# given a graph and a check node, returns the core value of the highest neighbor of the graph
def findClosestCore(communities, n, g):
    coreID = list()
    currentMax = -999
    ties = 0
    for c in range(0, len(communities)):
        temp = 0
        for u, v, dictt in g.edges(n, data=True):  # iterate through all the edges in n
            if n == u:  # v is the other node
                if (
                    v in communities[c]
                ):  # if the other node is in the community in question
                    temp += dictt["neighborOverlap"]  # update comm's temp score
            else:  # u is the other node
                if u in c:  # if the other node is in the community in question
                    temp += dictt["neighborOverlap"]  # update comm's temp score
        if (temp > currentMax) and not (c in coreID):
            currentMax = temp
            coreID.clear()
            ties = 0
            coreID.append(c)
        elif temp == currentMax:
            ties += 1
            coreID.append(c)
    if len(coreID) == 0:
        coreID.append(-1)
    return coreID


# assign the stragglers by number of connections, not quality of connections
def secondSort(g, straggler, communities):
    tally = np.array([0] * len(communities))
    neighIter = iter(nx.neighbors(g, straggler))
    while True:
        try:
            this = next(neighIter)
        except:
            break
        temp = g.nodes[this]["coreValue"]
        for core in range(0, len(temp)):
            mark = temp[core]
            if mark != -1:
                tally[mark] += 1
    if np.any(tally) > 0:
        # find the max and assign them to their max communities
        maxx = list()  # records the indexes of the max values
        maxx.append(-1)
        for m in range(0, len(tally)):
            if tally[m] > tally[maxx[0]]:
                maxx.clear()
                maxx.append(m)
            elif tally[m] == tally[maxx[0]]:
                maxx.append(m)
        return maxx
    return [-1]


# checks if any node is unassigned, what node it is
def areUnassigned(g):
    nodeIter = iter(g.nodes)
    count = 0
    numOverlapping = 0
    for i in range(0, len(g.nodes)):
        this = next(nodeIter)
        tempp = g.nodes[this]["coreValue"]
        if tempp[0] == -1:
            count += 1
        else:
            if len(tempp) > 1:
                numOverlapping += 1
    return None


# Function to convert communities lists to node dictionary - for Communities API
def convertToCommunityDictionary(communities):
    communityDictionary = dict()
    index = 0
    for comm in communities:
        for co in comm:
            communityDictionary.update({co: index})
        index += 1
    return communityDictionary
