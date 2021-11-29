from graphLib import Graph
from scipy.sparse import linalg
import scipy as sp
import scipy.sparse as ssp
import numpy as np
from collections import defaultdict 
from itertools import combinations
from operator import itemgetter


# =================== Task 1: Compute Node Centralities ===================
# ---------- Task 2.1: eigenvector centrality
def get_eigen_centrality(graph):
    # adj matrix of graph
    A = graph.get_adj_matrix()
    # change element in A into float type
    A_float = A.astype(float)

    # use linalg.eigs() to get eigenvalue and vectors
    eigenvalue, eigenvector = linalg.eigs(A_float, k=1, which='LR')
    eigenvector = eigenvector.real
    largest = eigenvector.flatten()

    # compute the norm
    norm = np.sign(largest.sum())*sp.linalg.norm(largest)
    normalized_eigenvector = eigenvector / norm
    # return the normalized eigen vector
    return normalized_eigenvector


# ------------ Task 2.2:  compute betweenness centrality
# compute the layers for a given source node s
def layers_BFS(graph, s):
    # this function is essentially the same as distance_BFS()
    # the only difference is that we need to store all the layers of node

    adj_list = graph.get_adj_list()
    visited = [False] * graph.n
    # store all layers of nodes
    Layers = []
    current_layer = [s]
    visited[s] = True
    next_layer = []
    depth = 0
    distance = np.zeros(graph.n, dtype = int)
    distance[s] = 0

    while len(current_layer)>0:
        for node in current_layer:
            for nbr in adj_list[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    next_layer.append(nbr)
        # update depth
        depth += 1
        # update distance for all the node in the next_level
        for i in next_layer:
            distance[i] = depth
        # add the current layers of nodes into Layers
        Layers.append(current_layer)
        # update current_layer and next_layer
        current_layer = next_layer
        next_layer = []
    return distance, Layers


# compute predecessors for all nodes other than s
def find_pred(graph, Layers, s):
    adj_list = graph.get_adj_list()
    # initialize a dictionary
    # key is the node id, value is a list of neighbors
    pred_dict = defaultdict(list)
    depth = len(Layers)

    # iterate from the second layer to the last layer
    for layer_num in range(1,depth):
        # nodes in the current layer
        current_layer = Layers[layer_num]
        # nodes in the previous layer
        pre_layer = Layers[layer_num-1]

        # find the predecessors for each node in current layer
        for node in current_layer:
            # predecessors are the intersect of neighbors and pre_layer
            nbrs = adj_list[node]
            pred_dict[node] = list(np.intersect1d(pre_layer, nbrs, assume_unique=True))
    return pred_dict


# find number of shortest paths from source node s to all other nodes
def find_num_shortest_path(graph, Layers, pred_dict, s):
    # store the number of shortest paths
    num_shortest_path = np.zeros(graph.n,dtype = int)
    num_shortest_path[s] = 1 # by default

    depth = len(Layers)
    # iterate from the second layer to the last layer
    for layer_num in range(1,depth):
        current_layer = Layers[layer_num]
        # calculate number of shortest paths from s to every node in the current layer
        for node in current_layer:
            # get the predecessors of node
            pred = pred_dict[node]
            for p_node in pred:
                num_shortest_path[node] += 1
    return num_shortest_path


# fill in the two matrices
def build_matrix(graph, Distance, Num_shortest_path):
    # treat each node in graph as the source node
    for s in range(graph.n):
        # get the distance and Layers
        distance, Layers = layers_BFS(graph, s)
        # fill in the s-th row of Distance matrix
        Distance[s, :] = distance

        # find the predecessors for all nodes
        pred_dict = find_pred(graph, Layers, s)
        # compute the number of shortest paths from s to all other nodes
        num_shortest_path = find_num_shortest_path(graph, Layers, pred_dict, s)
        # fill in the s-th row of Num_shortest_path
        Num_shortest_path[s, :] = num_shortest_path


# compute betweenness centrality for a node w, using the two matrices
def get_btw_c(w, Distance, Num_shortest_path):
    # all other nodes other than w
    rest_nodes = list(range(graph.n))
    rest_nodes.remove(w)
    # initialize centrality as 0
    btw_c = 0.0

    # enumerate each pair of nodes
    comb = combinations(rest_nodes, 2)
    comb_list = list(comb)
    for node_pair in comb_list:
        u = node_pair[0]
        v = node_pair[1]
        # get the distances for the three pairs of nodes (u,v), (w,v), (w,u)
        d_uv = Distance[u][v]
        d_wv = Distance[w][v]
        d_wu = Distance[w][u]
        if d_uv >= d_wu + d_wv: # check if w is on the shortest path of u and v
            # compute number of shortest path (nsp) for three pairs of nodes (u,v), (w,v), (w,u)
            nsp_uv = Num_shortest_path[u][v]
            nsp_wu = Num_shortest_path[w][u]
            nsp_wv = Num_shortest_path[w][v]
            # update btw_c
            btw_c += (nsp_wu * nsp_wv) / nsp_uv

    # return btw_c;
    # remember to normalize
    btw_c /= len(comb_list)
    return btw_c

'''
Main Function: Test all your functions here and perform any calculations 
'''
if __name__ == "__main__":
    # read the graph file and construct a graph
    graph = Graph(ssp.load_npz('../data/graph_BA.npz'))
    # TODO: find the top-10 nodes and their eigenvector centralities
    eigenvector = get_eigen_centrality(graph)
    eigenvector = eigenvector.T[0]
    indices = (-eigenvector).argsort()[:10]
    top_10_e_c = {}
    for index in indices:
        top_10_e_c[index] = eigenvector[index]
    print(top_10_e_c)

    # build 2 global matrices: a distance matrix, and a matrix storing the number of shortest paths
    n = graph.n
    Distance = np.zeros((n, n), dtype=int)
    Num_shortest_path = np.zeros((n, n), dtype=int)
    # construct the two matrices - it is recommended you save the two matrices into files since they
    # take a lot of time to compute
    build_matrix(graph, Distance, Num_shortest_path)

    # TODO: compute the betweenness centrality for each node
    btw_c = {}
    for w in range(n):
        btw_c[w] = get_btw_c(w, Distance, Num_shortest_path)
    top_10_btw_c = dict(sorted(btw_c.items(), key=itemgetter(1), reverse=True)[:10])
    print(top_10_btw_c)







