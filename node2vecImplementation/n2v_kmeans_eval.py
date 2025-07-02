import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import pickle

EMBEDDING_FILENAME = "n2v_node_embeddings.txt"
NUM_CLUSTERS = 3
NUM_NODES = 1000
ANNOTATION_OFFSET = 0.04

# Read in node embeddings
embeddings = pd.read_csv(EMBEDDING_FILENAME, sep=r'\s+', header=None)

# Instantiate a KMeans model
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=100)

# Fitting a KMeans model and getting the predicted labels (clusters) for each node
clusters = kmeans.fit_predict(embeddings.values)

# Creating a model to reduce the data to 2D using PCA (for plotting)
pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

# Obtaining the 2D node embeddings
pca_embeddings = pca_model.fit_transform(embeddings)

# Reading in the list of actual cluster labels (to compare the actual and predicted clusters)
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
node_blocks = sbm_data["Block/Cluster"]

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "..", "n2v_pca.png")

# Plot PCA of node embeddings
plt.scatter(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], c=node_blocks)
plt.savefig(output_path)

## Comparing KMeans classification with actual clusters

# Computing the normalized mutual info score (0 = no correlation, 1 = perfect correlation)
nmi_score = normalized_mutual_info_score(node_blocks, clusters)
print(f"NMI Score: {nmi_score}")