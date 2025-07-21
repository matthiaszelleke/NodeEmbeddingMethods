import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix

EMBEDDING_FILENAME = "lapeigen_node_embeddings.csv"
NUM_NODES = 300
NUM_CLUSTERS = 3
K = NUM_CLUSTERS # the number of dimensions for the node embeddings

## Generating node embeddings for the SBM

# Reading in the graph object
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
G = sbm_data["Graph"]
node_blocks = sbm_data["Block/Cluster"] # actual cluster labels, will be used to evaluate prediction performance

# Getting the graph's laplacian matrix L (L = D - A, D = diagonal degree matrix, A = adjacency matrix)
lap_matrix = nx.laplacian_matrix(G, weight=None)

# Above function returns a scipy spare matrix, converting to a numpy array to make sure entries are stored as ints
lap_matrix = lil_matrix(lap_matrix).toarray()

# Obtaining the eigenvalues and eigenvectors of the laplacian 
eigenvalues, eigenvectors = eigh(lap_matrix) # eigh() returns eigenvals in asc order

# extracting the k smallest (excluding the smallest) eigenvectors
k_smallest_eigv = eigenvectors[:, 1:K+1]

# Each row in the above created matrix is a K-dim embedding for node i
embeddings = k_smallest_eigv

# Saving embeddings in a csv file
np.savetxt("SpectralClusteringImplementation/" + EMBEDDING_FILENAME, embeddings, delimiter=',', fmt='%.6f')