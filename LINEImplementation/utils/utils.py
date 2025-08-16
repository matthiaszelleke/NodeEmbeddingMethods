# Copied from: https://github.com/dmpierre/LINE
# Copyright (c) 2025 Pierre Daix-Moreux
# Licensed under the MIT License (see LICENSE or LICENSE-pierre in this repository)
# Implements the LINE algorithm from:
# Tang et al. (2015). "LINE: Large-scale Information Network Embedding"

from tqdm import tqdm
from decimal import *
import random
import collections
import numpy as np

class VoseAlias(object):
    '''Implementation of the Vose Alias 
        Method for sampling'''

    def __init__(self, dist):
        self.dist = dist
        self.alias_initialization()

    def alias_initialization(self):
        '''Construct the alias table for the
            probability table'''
        n = len(self.dist)

        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        '''The rest of this method implements the Vose Alias method,
            see https://www.youtube.com/watch?v=retAwpUv42E
            for an explanation of it'''
        
        # Construct and sort the scaled probabilities into their appropriate stacks
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)
        
        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        # doing the sampling
        edge = random.choice(self.listprobs)
        if self.table_prob[edge] >= random.uniform(0, 1):
            return edge
        else:
            return self.table_alias[edge]
        
    def sample_n(self, size):
        for i in range(size):
            yield self.alias_generation()

def makeDist(network, power=0.75):
    '''Making the node and edge probability distributions,
        each of which are either scaled versions or functions
        of the degree distribution and weights distrubtion,
        respectively'''

    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0

    edges = network.edges.data("weight", default=1)
    for edge in edges:
        node1, node2, weight = edge[0], edge[1], edge[2]

        edgedistdict[tuple([node1, node2])] = weight # store weight for this edge
        nodedistdict[node1] += weight # update the outdegree of node1

        weightsdict[tuple([node1, node2])] = weight # store weight for this edge
        nodedegrees[node1] += weight # update the outdegree of node1

        weightsum += weight

        # negprobsum will be used to modify the node distribution
        negprobsum += np.power(weight, power)

    """
    with open(graphpath, "r") as graphfile:
        for l in graphfile:
            nlines += 1

    print("Reading edgelist file...")
    maxindex = 0
    with open(graphpath, "r") as graphfile:
        for l in tqdm(graphfile, total=nlines): # looping over each edge

            line = [int(i) for i in l.replace("\n", "").split(" ")]
            node1, node2, weight = line[0], line[1], line[2]

            edgedistdict[tuple([node1, node2])] = weight # store weight for this edge
            nodedistdict[node1] += weight # update the outdegree of node1

            weightsdict[tuple([node1, node2])] = weight # store weight for this edge
            nodedegrees[node1] += weight # update the outdegree of node1

            weightsum += weight

            # negprobsum will be used to modify the node distribution
            negprobsum += np.power(weight, power)

            if node1 > maxindex:
                maxindex = node1
            elif node2 > maxindex:
                maxindex = node2

    """
    for node, outdegree in nodedistdict.items():
        # in addition to scaling the degree, we raise each node to a power < 1
        # to prevent nodes with high outdegrees from being sampled too often
        nodedistdict[node] = np.power(outdegree, power) / negprobsum
    
    for edge, weight in edgedistdict.items():
        # scale edge weight to represent a probability in a prob. distribution
        edgedistdict[edge] = weight / weightsum

    return edgedistdict, nodedistdict

def negSampleBatch(sourcenode, targetnode, negsamplesize, nodealiassampler):
    '''Sample negative nodes for a given pair of 
        positive nodes (nodes connected by an edge)'''
    negsamples = 0
    while negsamples < negsamplesize:
        # nodealiassampler is an object which does the Vose Alias sampling
        sampled_nodes = nodealiassampler.sample_n(1)
        sampled_node = next(iter(sampled_nodes))
        if (sampled_node == sourcenode) or (sampled_node == targetnode):
            continue
        else:
            negsamples += 1
            yield sampled_node

def makeData(samplededges, negsamplesize, nodesaliassampler):
    '''Essentially a wrapper to negSampleBatch'''
    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = list(negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        nodesaliassampler))
        yield [e[0], e[1]] + negnodes
                                    