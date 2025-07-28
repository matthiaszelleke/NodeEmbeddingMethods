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

NUM_NODES = 100

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

# Obtaining the integer representation of each position/cluster (to index into connectivity matrix)
node_cluster = np.zeros(shape=NUM_NODES, dtype=int)
for i, position in enumerate(actual_positions):
    node_cluster[i] = enumerate_positions[actual_positions[i]]

# Using the connectivity matrix to produce the graph's edges 
for i in range(NUM_NODES):
    for j in range(i):
        # Getting the probability of an edge connection between node i and j
        connection_prob = connectivity_matrix[node_cluster[i], node_cluster[j]]
        has_edge = bernoulli.rvs(connection_prob) # Sampling a bernoulli distribution with the predefined connection probability
        if has_edge:
            # Adding an edge if a success (1) was sampled
            G.add_edge(i+1, j+1) # adding 1 since node labels in graph start from 1, not 0

clusters = [] # A 2D list which will store all the same cluster labels in the same row
# Adding all the nodes in the same cluster to the 2D list, cluster by cluster
for i in range(len(positions)):
    this_cluster = node_cluster[node_cluster == i] # filtering for all nodes in clustering i
    clusters.append(this_cluster)

# A 2D list which will store all the nodes with the same cluster label in the same row
node_list = [[] for _ in range(len(positions))]
for node_id in G.nodes:
    cluster = node_cluster[node_id - 1]  # node_id starts from 1
    node_list[cluster].append(node_id)


## Plotting an SBM

# Randomly choosing only 100 of the nodes to be plotted, so the plot isn't too crowded
drawn_nodes = random.sample(range(1, NUM_NODES+1), 100)
drawn_nodes = np.array(drawn_nodes, dtype=np.int32)

drawn_node_cluster = node_cluster[drawn_nodes-1]  # Only keep the cluster labels for the chosen nodes

# The subgraph of the original graph induced by the chosen nodes
drawn_G = nx.induced_subgraph(G, drawn_nodes)


# Obtaining plotting coordinates for each node to be drawn
cluster_centers = [(-5, 5), (5, 5), (0, -5)] # Coordinates of cluster centres (for plotting)
pos = {}
# Iterate over all clusters
for cluster, nodes_in_cluster in enumerate(node_list):

    # Extracting the nodes in this cluster which were chosen for drawing
    cluster_drawn_nodes = [n for n in nodes_in_cluster if n in drawn_nodes]
    
    if not cluster_drawn_nodes:  # The edge case where no nodes in this cluster were chosen for drawing
        continue

    # Subgraph with only these nodes
    cluster_drawn_G = drawn_G.subgraph(cluster_drawn_nodes)

    # Generate the subpositions for each node ("noise" position added to cluster center above)
    sub_pos = nx.spring_layout(cluster_drawn_G, center=(0, 0), scale=3.0, k=1)

    # Offset positions by the cluster center
    center = np.array(cluster_centers[cluster])
    for node, noise in sub_pos.items():
        pos[node] = center + noise

# Getting the cluster label (used for colouring) for each node to be drawn
drawn_node_colors = [node_cluster[node_id - 1] for node_id in drawn_G.nodes()]

# Draw this smaller network of 100 nodes and their edges, colouring nodes by their cluster
nx.draw_networkx(drawn_G, pos, node_size=30, node_color=drawn_node_colors, cmap=plt.cm.tab10, 
                 width=0.5, alpha=0.5, with_labels=False)

plt.title("SBM network: Nodes colored by cluster")
plt.savefig("sbm.png", dpi=300)

# Saving the list of cluster membership (0, 1, or 2) of each node
with open("sbm_actual_labels.pkl", "wb") as f:
    pickle.dump({"Graph": G, "Block/Cluster": node_cluster}, f)

# Saving the graph
nx.write_edgelist(G, "sbm_graph.edgelist", data=False)

# Specifying the edge-weight to be 1 for every edge
with open("sbm_graph.edgelist", "w") as f:
    for u, v in G.edges():
        f.write(f"{u} {v} 1\n")