import numpy as np
import scipy.sparse as ssp
from graphLib import Graph
from itertools import combinations
import matplotlib.pyplot as plt
from collections import defaultdict


# =================== Task 1: Compute Graph Properties ===================

# ------------- Task 1.1: compute degree distribution
def compute_deg_distr(graph):
    # compute the degree distribution of the input graph
    # return two array's
    # degs: all the possible node degrees, in ascending order
    # counts: the corresponding counts (number of nodes) for each degree value in degs

    # get the degree sequence of all the nodes
    deg_seq = graph.get_deg_seq()

    # get degree and its corresponding count
    # using np.unique()
    degs, counts = np.unique(deg_seq, return_counts=True)
    return degs, counts


# ------------ Task 1.2: compute clustering coefficient
def get_cc_local(graph,nid):

    # adj matrix of graph in dense form
    adj_matrix = graph.get_adj_matrix()
    # adj_list of graph
    adj_list = graph.get_adj_list()

    # get the neighbors of node nid
    nbrs = adj_list[nid]
    # number of neighbors of node nid
    nbrs_num = len(nbrs)

    # number of connected pairs of neighbors
    closed_pairs = 0

    if nbrs_num < 2:
        return 0.0
    else:
        # enumerate all pairs of neighbors
        comb = combinations(nbrs, 2)
        comb_list = list(comb)
        for node_pair in comb_list:
            # check if this pair of nodes are connected
            if node_pair[1] in adj_list[node_pair[0]]:
                closed_pairs += 1

        # degree of node nid
        deg = nbrs_num
        return closed_pairs / len(comb_list)


def get_cc_global(graph):
    adj_matrix = graph.get_adj_matrix()
    adj_list = graph.get_adj_list()
    # number of nodes in graph
    n = graph.n

    # number of closed 2-pth
    closed_2_path = 0
    # number of 2-path
    all_2_path = 0

    # loop over all nodes
    for i in range(n):
        # get the neighbors of node i
        nbrs = adj_list[i]
        nbrs_num = len(nbrs)
        if nbrs_num >= 2:
            # enumerate all pairs of neighbors
            comb = combinations(nbrs, 2)
            for node_pair in list(comb):
                # update all_2_path
                all_2_path += 1
                # update closed_2_path
                if node_pair[1] in adj_list[node_pair[0]]:
                    closed_2_path += 1

    return closed_2_path / all_2_path


# --------------------- Task 1.3: compute diameter
def distance_BFS(graph, s):
    adj_list = graph.get_adj_list()
    # return the distance of s to every node
    # travel the graph from s using BFS; this is will create a tree rooted at s
    # the hight of the tree is the longest distance

    # use a vector to record if a node is visited or not
    visited = [False] * graph.n
    visited[s] = True
    # store the distance from s to all other nodes
    distance = np.zeros(graph.n, dtype = int)
    distance[s] = 0
    # current layer of nodes
    current_level = [s]
    # next layer of nodes
    next_level = []
    # the number of layers
    depth = 0

    # while current layer is not empty
    while len(current_level) > 0:
        for node in current_level: # each node in current layer
            for nbr in adj_list[node]: # for each neighbor of this node
                # add a neighbor into next layer, if the neighbor is not visited yet
                # remember to update visited
                if not visited[nbr]:
                    visited[nbr] = True
                    next_level.append(nbr)

        # update depth
        depth += 1
        # update distance from s to all nodes in the next_level
        for child in next_level:
            distance[child] = depth
        # set current_level as next_level
        current_level = next_level
        # empty next_level
        next_level = []

    return distance


def get_diameter(graph):
    adj_list = graph.get_adj_list()
    diameter = 0

    # treat every node in graph as the source node
    # call distance_BFS() to get the distance
    # find the max distance
    for source in range(graph.n):
        distance = distance_BFS(graph, source)
        diameter = max(diameter, max(distance))
    return diameter

'''
Main Function: Test all your functions here and perform any calculations 
'''
if __name__ == "__main__":
    # --- read the graph file and construct a graph
    graph_ER = Graph(ssp.load_npz('../data/graph_ER.npz'))
    graph_BA = Graph(ssp.load_npz('../data/graph_BA.npz'))

    # TASK 1.1
    degs, counts = compute_deg_distr(graph_ER)
    distribution = counts / graph_ER.n
    plt.bar(degs, distribution)
    plt.xlabel('Node degree')
    plt.ylabel('Frequency $P(k) = n_k/n$')
    plt.show()

    degs, counts = compute_deg_distr(graph_BA)
    distribution = counts / graph_BA.n
    plt.bar(degs, distribution)
    plt.xlabel('Node degree')
    plt.ylabel('Frequency $P(k) = n_k/n$')
    plt.show()

    # TASK 1.2
    global_cc_graph_1 = get_cc_global(graph_ER)
    global_cc_graph_2 = get_cc_global(graph_BA)
    print("global cc for graph 1: {}".format(global_cc_graph_1))
    print("global cc for graph 2: {}".format(global_cc_graph_2))

    nodes = [0, 11, 24, 50, 150, 250, 334, 668]
    for node in nodes:
        local_cc = get_cc_local(graph_ER, node)
        print("local cc for graph 1, node {}: {}".format(node, local_cc))
    for node in nodes:
        local_cc = get_cc_local(graph_BA, node)
        print("local cc for graph 2, node {}: {}".format(node, local_cc))

    # TASK 1.3
    diameter_graph_1 = get_diameter(graph_ER)
    diameter_graph_2 = get_diameter(graph_BA)
    print("diameter of graph 1: {}".format(diameter_graph_1))
    print("diameter of graph 2: {}".format(diameter_graph_2))

    # Test the rest of your functions here using your graph  


