import networkx as nx
import numpy as np
import random
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import pickle
from sbm_config import args

## Simulating an SBM

# Create empty graph
G = nx.Graph()

NUM_NODES = 1000

# Possible positions/roles for someone at the university
positions = ["undergraduate", "graduate", "faculty"]

# A convenient mapping from positions to integers
enumerate_positions = {"undergraduate":0, "graduate":1, "faculty":2}

# Probability of a person at the university being having each position 
position_probs = [1/3, 1/3, 1/3]

# Sampling a position for each person
actual_positions = random.choices(positions, position_probs, k=NUM_NODES)

# Sorting the positions (with undergraduate < graduate < faculty) to enable comparison 
# with predicted positions/clusters later
actual_positions.sort(key = lambda position: enumerate_positions[position])

# Adding a node for each person, along with their position
for node in range(NUM_NODES):
    G.add_node(node+1, position = actual_positions[node])

# The matrix defining the probability of a person in position i
# and a person in position j being friends (having an edge between them)
connectivity_matrix = np.array([[args.probability1, 0.1, 0.1],
                                [0.1, args.probability2, 0.1],
                                [0.1, 0.1, args.probability3]
                      ])

# Obtaining the integer representation of each position (to index into connectivity matrix)
node_blocks = np.zeros(shape=NUM_NODES, dtype=int)
for i, position in enumerate(actual_positions):
    node_blocks[i] = enumerate_positions[actual_positions[i]]

# Using the connectivity matrix to produce the graph's edges 
for i in range(NUM_NODES):
    for j in range(i):
        # Getting the probability of an edge connection between node i and j
        connection_prob = connectivity_matrix[node_blocks[i], node_blocks[j]]
        has_edge = bernoulli.rvs(connection_prob) # Sampling a bernoulli distribution with the predefined connection probability
        if has_edge:
            # Adding an edge if a success (1) was sampled
            G.add_edge(i+1, j+1) # adding 1 since node labels in graph start from 1, not 0

clusters = [] # A 2D list which will store nodes in the same cluster in their own row
# Adding all the nodes in the same cluster to the 2D list, cluster by cluster
for i in range(len(positions)):
    this_cluster = node_blocks[node_blocks == i] # filtering for all nodes in clustering i
    clusters.append(this_cluster)

node_list = [] # A 2D list, with the same shape as the 2D list clusters, where each entry is the node's label/ID
first_ID = 1 # A counter variable to keep track of node IDs across different clusters
for cluster in clusters:
    this_list = list(range(first_ID, first_ID+len(cluster))) # list of node IDs for nodes in "this" cluster
    node_list.append(this_list)

    first_ID = first_ID+len(cluster) # update firstID to be the ID of the first node in the next cluster


## Plotting an SBM

cluster_centers = [(-5, 5), (5, 5), (0, -5)] # Coordinates of cluster centres

pos = {}
# Iterate over all clusters
for cluster, nodes in enumerate(node_list):
    # Subgraph with only nodes in this cluster
    cluster_G = G.subgraph(nodes)

    # Generate the subpositions for each node ("noise" position added to cluster position)
    sub_pos = nx.spring_layout(cluster_G, center=(0, 0), scale=1.0, k=3)

    # Offset positions by the cluster center
    offset = np.array(cluster_centers[cluster])
    for node, coord in sub_pos.items():
        pos[node] = coord + offset

# Create a mapping from positions to colours, for plotting
node_colours = ['tab:blue', 'tab:orange', 'tab:green']

# Plot nodes, one cluster at a time
for nodeIDs, clr in zip(node_list, node_colours):
    nx.draw_networkx_nodes(G, pos, nodelist=nodeIDs, node_color=clr, node_size=50)

# Plot edges
nx.draw_networkx_edges(G, pos)

plt.title("Clustered Layout: Nodes Positioned by Cluster")
plt.axis("off")

# Saving the plot
plt.savefig("sbm.png")

# Saving the list of cluster membership (0, 1, or 2) of each node
with open("sbm_actual_labels.pkl", "wb") as f:
    pickle.dump({"Graph": G, "Block/Cluster": node_blocks}, f)

# Saving the graph
nx.write_edgelist(G, "sbm_graph.edgelist", data=False)

# Specifying the edge-weight to be 1 for every edge
with open("sbm_graph.edgelist", "w") as f:
    for u, v in G.edges():
        f.write(f"{u} {v} 1\n")
