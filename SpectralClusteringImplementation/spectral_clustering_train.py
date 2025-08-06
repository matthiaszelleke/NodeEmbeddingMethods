from .config import args
import networkx as nx
import numpy as np
from numpy.linalg import eigh
from utils import get_node_clusters, get_num_clusters
from scipy.sparse import lil_matrix


def get_embeddings_spectral_clustering(network, embeddings_filename="spectral_clustering_node_embeddings.csv"):

    ## Generating node embeddings for the SBM

    node_clusters = get_node_clusters(network)
    num_clusters = get_num_clusters(node_clusters)
    K = num_clusters

    # Getting the graph's laplacian matrix L (L = D - A, D = diagonal degree matrix, A = adjacency matrix)
    lap_matrix = nx.laplacian_matrix(network, weight=None)

    # Above function returns a scipy spare matrix, converting to a numpy array to make sure entries are stored as ints
    lap_matrix = lil_matrix(lap_matrix).toarray()

    # Obtaining the eigenvectors of the laplacian 
    _, eigenvectors = eigh(lap_matrix)

    # extracting the k smallest (excluding the smallest) eigenvectors
    k_smallest_eigv = eigenvectors[:, 1:K+1]

    # Each row in the above created matrix is a K-dim embedding for node i
    embeddings = k_smallest_eigv

    # Saving embeddings in a csv file
    np.savetxt("SpectralClusteringImplementation/"+embeddings_filename, embeddings, delimiter=',', fmt='%.6f')

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network, nodetype=int)
    get_embeddings_spectral_clustering(network, args.emb_fname)