import numpy as np
import scipy.sparse as ssp
from graphLib import Graph
import matplotlib.pyplot as plt
from scipy import sparse
import math

# =================== Task 5: Inuence Maximization and Blocking ===================

# parameters
p = 0.05
mc = 5000
k = 10


# ------------------------- Task 5.1: influence maximization
# simulate a single diffusion process
def info_diff(adj_list, seed_set, p):
    # seed_set: the initial set of seeds
    # p: universal activation probability 
    # output: the number of activated nodes
    n = len(adj_list)
    if len(seed_set) == 0:
        return 0
    # store the active_nodes   
    active_nodes = list(seed_set)
    # store currently active nodes
    current_active = list(seed_set)
    # active nodes are flagged as 1
    activated_flag = np.zeros(n, dtype=int)
    activated_flag[seed_set] = 1

    while len(current_active) > 0:
        # use the nodes in current_active to activate others
        newly_activated = []
        for u in current_active: # each node in current_active has the chance to activate neighbors
            # neighbors of nbrs
            nbrs = adj_list[u]
            for v in nbrs: # add v into newly_activated according to the model
                rnd = np.random.uniform(0.0, 1.0)
                if rnd >= p and activated_flag[v] != 1:
                    newly_activated.append(v)
                    activated_flag[v] = 1

        # add newly activated nodes into active nodes
        active_nodes.extend(newly_activated)
        # update current active nodes
        current_active = newly_activated
    return len(active_nodes)

# estimate the influence of a set of seeds
def get_influence(graph, seed_set, p, mc):
    # seed_set: a given set of seeds
    # mc: simulation times
    np.random.seed(14)
    
    # initialize the total number of active nodes throughout simulations
    total_active_num = 0
    adj_list = graph.get_adj_list()
    for i in range(mc):
        # run a single simulation; update total_active_num
        active_num = info_diff(adj_list, seed_set, p)
        total_active_num += active_num
    return total_active_num/mc


# find the optimal set of seeds using greedy search
def greedySearch(graph, k, p, mc):
    n = graph.n
    # find seeds in a candidate set
    # initially, candidate set contains all the nodes
    candidate = list(range(n))
    seed_set = [] # store the seeds

    influence = [0.0] # store the influence of each set of seeds
    for i in range(k): # repeat k times to find k seeds
        # influence of current set of seeds
        current_influence = influence[i]

        most_influence_node = 0
        most_influence = 0
        # --- find most influential node in the candidate set
        # influence of a node = influence of ( current_set + node) -  influence of current_set
        for u in candidate:
            u_influence = get_influence(graph, seed_set+[u], p, mc)
            if most_influence - current_influence < u_influence - current_influence:
                most_influence_node = u
                most_influence = u_influence

        # most_influence_node is the node with the highest influence
        # add it to the seed_set
        seed_set.append(most_influence_node)
        # store the influence of current set of seeds
        influence.append(get_influence(graph, seed_set, p, mc))
        # remove most_influence_node from candidate set
        candidate = np.setdiff1d(candidate, most_influence_node)

        print("done with iteration {}".format(i))
    return seed_set, influence


# ----------------- Task 5.2: random blocking ------------
# modify the graph by deleting edges
def modify_graph(graph, target_nodes):
    adj_list = graph.get_adj_list()
    A = graph.get_adj_matrix()
    np.random.seed(13)
    # construct the adj matrix of a new graph
    A_new = A.copy()

    for u in target_nodes:
        nbrs = adj_list[u]
        # randomly select a neighbor
        v = np.random.choice(nbrs, 1)
        # delete the edge; remember to modify two entries
        A_new[u, v] = 0
        A_new[v, u] = 0

    # transfer to scipy.sparse.csr.csr_matrix
    return sparse.csr_matrix(A_new)

'''
Main Function: Test all your functions here and perform any calculations 
'''
if __name__ == "__main__":
    graph = Graph(ssp.load_npz('../data/BA_graph_100.npz'))
    # seeds, influence = greedySearch(graph, k, p, mc)
    #
    # # we recommend saving the results into files
    # np.savetxt('../result/IM_seeds.txt', seeds, fmt='%s')
    # np.savetxt('../result/IM_influence.txt', influence, fmt='%f')

    # uncomment these lines if you want to read the values instead of recalculating them
    seeds = np.loadtxt('../result/IM_seeds.txt', dtype = int)
    influence = np.loadtxt('../result/IM_influence.txt', dtype = float)

    # randomly select 50 nodes
    np.random.seed(15)
    target_nodes = np.random.choice(range(graph.n), 50, replace=False)

    # each node in seed_set will randomly block a neighbor
    # equivalently, delete the edge between that node and the neighbor

    # create a modified graph
    graph_new = Graph(modify_graph(graph, target_nodes))

    # TODO: compute the seeds and influences over the new graph using the same parameters
    # seeds_new, influence_new = greedySearch(graph_new, k, p, mc)
    #
    # np.savetxt('../result/IM_seeds_new.txt', seeds_new, fmt='%s')
    # np.savetxt('../result/IM_influence_new.txt', influence_new, fmt='%f')

    # uncomment these lines if you want to read the values instead of recalculating them
    seeds_new = np.loadtxt('../result/IM_seeds_new.txt', dtype = int)
    influence_new = np.loadtxt('../result/IM_influence_new.txt', dtype = float)

    # plot in the same figure
    plt.plot(range(k+1), influence, '-bo')
    plt.show()
    plt.plot(range(k+1), influence_new, '-rs')
    plt.show()




