import numpy as np
import os
from config import args
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import pickle

EMBEDDINGS_FILENAME = args.embeddings_file
NUM_CLUSTERS = 3
NUM_NODES = 1000
ANNOTATION_OFFSET = 0.04

embeddings = np.genfromtxt(EMBEDDINGS_FILENAME, delimiter=",", skip_header=1)

# Instantiate a KMeans model
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=100)

# Fitting a KMeans model and getting the predicted labels (clusters) for each node
clusters = kmeans.fit_predict(embeddings)

# Performing 2D PCA on the node embeddings
pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)
pca_embeddings = pca_model.fit_transform(embeddings)

# Plotting the PCA node embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "..", "grarep_pca.png")

plt.scatter(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1])
plt.savefig(output_path)

## Comparing KMeans classification with actual clusters

# Reading in the list of actual cluster labels
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
node_blocks = sbm_data["Block/Cluster"]

# Computing the normalized mutual info score (0 = no correlation, 1 = perfect correlation)
nmi_score = normalized_mutual_info_score(node_blocks, clusters)
print(f"NMI Score: {nmi_score}")
