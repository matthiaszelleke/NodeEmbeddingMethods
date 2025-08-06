import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from sbm_config import args
from utils import get_node_clusters, get_num_clusters


def get_cluster_centers(num_clusters, radius):
    # Get the coordinates for the center of each cluster when plotted

    cluster_centers = []
    
    for cluster in range(num_clusters):
        angle = cluster * (2*math.pi/num_clusters)
        center_x = radius*math.cos(angle)
        center_y = radius*math.sin(angle)
        cluster_centers.append((center_x, center_y))

    return cluster_centers


def plot_network(network):

    ## Plotting an SBM

    num_nodes = network.number_of_nodes()
    node_clusters = get_node_clusters(network)
    num_clusters = get_num_clusters(node_clusters)

    clusters_unique = list(set(node_clusters))
    enumerate_clusters = {cluster: cluster_id for cluster_id, cluster in enumerate(clusters_unique)}

    # A 2D list which will store all the nodes with the same cluster label in the same row
    node_list = [[] for _ in range(num_clusters)]
    for node_id in network.nodes:
        cluster = node_clusters[node_id - 1]  # node_id starts from 1
        cluster_id = enumerate_clusters[cluster] # getting the int representation (cluster id) of the cluster

        node_list[cluster_id].append(node_id)
    
    if num_nodes > 100:
        # Randomly choosing only 100 of the nodes to be plotted, so the plot isn't too crowded
        drawn_nodes = random.sample(range(1, num_nodes+1), 100)
        drawn_nodes = np.array(drawn_nodes, dtype=np.int32)
    else:
        drawn_nodes = np.arange(1, num_nodes+1)

    # The subgraph of the original graph induced by the chosen nodes
    drawn_G = nx.induced_subgraph(network, drawn_nodes)

    # Obtaining plotting coordinates for each node to be drawn
    cluster_centers = get_cluster_centers(num_clusters, radius=7)
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

    # Getting the cluster ids (used for colouring) for each node to be drawn
    drawn_cluster_ids = [enumerate_clusters[node_clusters[node_id - 1]] for node_id in drawn_G.nodes()]

    # Draw this smaller network of 100 nodes and their edges, colouring nodes by their cluster
    nx.draw_networkx(drawn_G, pos, node_size=30, node_color=drawn_cluster_ids, cmap=plt.cm.tab10, 
                    width=0.5, alpha=0.5, with_labels=False)

    plt.title("Network: nodes coloured by cluster")
    plt.savefig("network.png", dpi=300)

if __name__ == "__main__":
    network = nx.read_edgelist(args.network_fname)
    plot_network(network)