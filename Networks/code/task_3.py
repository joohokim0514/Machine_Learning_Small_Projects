
from graphLib import Graph
from scipy.sparse import linalg
import scipy as sp
import scipy.sparse as ssp
import numpy as np
from collections import defaultdict 
from itertools import combinations 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv

# =================== Task 3: link prediction ===================
# generate an observed network
def gen_net_obs(graph, test_edges, test_non_edges):
    adj_matrix = graph.get_adj_matrix()
    # adjacency matrix of the observed graph, which is a copy of the adj_matrix of the original graph
    adj_obs = adj_matrix.copy()
    # number of edges in target set
    pos_num = len(test_edges)
    # number of non_edges in target set
    neg_num = len(test_non_edges)
    # test_set is the union of test edges and non-edges
    test_set = np.concatenate((test_edges, test_non_edges), axis=0)
    # --remove edges by changing adj_obs
    for edge in test_edges:
        adj_obs[edge[0], edge[1]] = 0
        adj_obs[edge[1], edge[0]] = 0
    return adj_obs, test_set


# compute Jaccard similarity for all node pairs
def compute_Jaccard(adj, node_pairs): # cn/d_u + d_v - cn
    # adj is the adjacency matrix of the observed network
    pair_num = node_pairs.shape[0]
    # store the similarity scores for all the node pairs
    sim_vec = np.zeros(pair_num, dtype=float)

    for i in range(pair_num):
        u = node_pairs[i, 0]
        v = node_pairs[i, 1]
        # number of common neighbors
        cn_num = len(np.intersect1d(np.where(adj[u, :] == 1), np.where(adj[v, :] == 1)))
        # degree of u and v
        deg_u = np.sum(adj[u, :])
        deg_v = np.sum(adj[v, :])
        # similarity between u and v
        sim_vec[i] = cn_num / (deg_u+deg_v-cn_num)
    return sim_vec


# compute Katz similarity for all node pairs
def compute_Katz(adj, node_pairs):
    # adj is the adjacency matrix of the observed network
    pair_num = node_pairs.shape[0]
    sim_vec = np.zeros(pair_num,dtype = float)
    n = adj.shape[0]

    #compute similarity matrix
    beta = 0.09
    I = np.identity(n)
    Katz_matrix = np.linalg.inv(I - beta*adj) - I
    # get the Katz similarity for each pair of node
    for i in range(pair_num):
        sim_vec[i] = Katz_matrix[node_pairs[i,0], node_pairs[i,1]]
    return sim_vec


# link prediction, by comparing similarity with a threshold theta
def link_pred(adj, node_pairs, metric, theta):
    # adj is the adjacency matrix of the observed network
    # node_pairs are all the pairs of node
    # metric = 'Jaccard' or 'Katz'
    # theta is the threshold
    # first, compute the similarity
    if metric == 'Jaccard':
        sim_vec = compute_Jaccard(adj,node_pairs)
    elif metric == 'Katz':
        sim_vec = compute_Katz(adj,node_pairs)

    # make prediction
    pred = np.zeros(len(node_pairs), dtype=int)

    # if sim_vec[i] >= theta, predict node pair i as a link (set pred[i] as 1)
    # otherwise set pred[i] as 0
    pred = np.where(sim_vec >= theta, 1, 0)
    return pred

'''
Main Function: Test all your functions here and perform any calculations 
'''
if __name__ == "__main__":
    graph = Graph(ssp.load_npz('../data/ws_graph.npz'))
    # load the test edges and non_edges
    test_edges = np.loadtxt('../data/target.edges',dtype = int)
    test_non_edges = np.loadtxt('../data/target.nonedges', dtype = int)
    pos_num = len(test_edges)
    neg_num = len(test_non_edges)
    # 1 denotes an edge; 0 denotes a non-edge
    ground_truth = np.concatenate((np.ones(pos_num, dtype=int), np.zeros(neg_num, dtype=int)), axis=0)
    # TODO: generate observed network
    adj_obs, test_set = gen_net_obs(graph, test_edges, test_non_edges)

    # TODO: Get similarity based on metric (Katz or Jaccard)
    metric = 'Jaccard'
    sim_vec = compute_Jaccard(adj_obs, test_set)
    # save the similarity scores; remember to change the file name
    np.savetxt('../result/Jac_sim.txt', sim_vec, fmt = '%f')
    # TODO: make prediction for Jaccard
    pred = link_pred(adj_obs, test_set, metric, 0.1)
    # TODO: compute accuracy based on pred and groundtruth
    # accuracy  =  correctly predicted node pairs/total number of pairs
    acc = np.count_nonzero(np.equal(ground_truth, pred)) / len(ground_truth)
    print("Jaccard accuracy: {}".format(acc))

    # TODO: Get similarity based on metric (Katz or Jaccard)
    metric = 'Katz'
    sim_vec = compute_Katz(adj_obs, test_set)
    # save the similarity scores; remember to change the file name
    np.savetxt('../result/Katz_sim.txt', sim_vec, fmt='%f')
    # TODO: make prediction for Katz
    pred = link_pred(adj_obs, test_set, metric, 0.02)
    # TODO: compute accuracy based on pred and groundtruth
    # accuracy  =  correctly predicted node pairs/total number of pairs
    acc = np.count_nonzero(np.equal(ground_truth, pred)) / len(ground_truth)
    print("Katz accuracy: {}".format(acc))
