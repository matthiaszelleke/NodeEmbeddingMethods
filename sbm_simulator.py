import networkx as nx
import numpy as np
import random
from sbm_config import args
from scipy.stats import bernoulli
import pickle


def generate_sbm_network(num_nodes, p1, p2, p3):

    ## Simulating an SBM

    # Create empty graph
    G = nx.Graph()

    # Possible positions/roles for someone at the university
    positions = ["undergraduate", "graduate", "faculty"]

    # A convenient mapping from positions to integers
    enumerate_positions = {"undergraduate":0, "graduate":1, "faculty":2}

    # Probability of a person at the university being having each position 
    position_probs = [1/3, 1/3, 1/3]

    # Sampling a position for each person
    actual_positions = random.choices(positions, position_probs, k=num_nodes)

    # Sorting the positions (with undergraduate < graduate < faculty) to enable comparison 
    # with predicted positions/clusters later
    actual_positions.sort(key = lambda position: enumerate_positions[position])

    # Adding a node for each person, along with their position
    for node in range(num_nodes):
        G.add_node(node+1, label = actual_positions[node])

    # The matrix defining the probability of a person in position i
    # and a person in position j being friends (having an edge between them)
    connectivity_matrix = np.array([[p1, 0.1, 0.1],
                                    [0.1, p2, 0.1],
                                    [0.1, 0.1, p3]
                        ])

    # Obtaining the integer representation of each position/cluster (to index into connectivity matrix)
    node_cluster = np.zeros(shape=num_nodes, dtype=int)
    for i, position in enumerate(actual_positions):
        node_cluster[i] = enumerate_positions[actual_positions[i]]

    # Using the connectivity matrix to produce the graph's edges 
    for i in range(num_nodes):
        for j in range(i):
            # Getting the probability of an edge connection between node i and j
            connection_prob = connectivity_matrix[node_cluster[i], node_cluster[j]]
            has_edge = bernoulli.rvs(connection_prob) # Sampling a bernoulli distribution with the predefined connection probability
            if has_edge:
                # Adding an edge if a success (1) was sampled
                G.add_edge(i+1, j+1) # adding 1 since node labels in graph start from 1, not 0

    
    # Saving the list of cluster membership (0, 1, or 2) of each node
    with open("sbm_actual_labels.pkl", "wb") as f:
        pickle.dump({"Block/Cluster": node_cluster}, f)

    # Saving the graph
    nx.write_edgelist(G, "sbm_graph.edgelist", data=False)

    # Specifying the edge-weight to be 1 for every edge
    with open("sbm_graph.edgelist", "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v} 1\n")
    
    return G

if __name__ == "__main__":
    generate_sbm_network(args.num_nodes, args.p1, args.p2, args.p3)