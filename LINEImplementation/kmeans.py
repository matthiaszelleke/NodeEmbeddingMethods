import os
import torch
from config import args
from utils.line import LINE
from train import args
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import pickle

MODEL_FILENAME = "LINE_model.pt"
NUM_CLUSTERS = 3
NUM_NODES = 1000
ANNOTATION_OFFSET = 0.04

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiating empty LINE models
line_ord1 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)
line_ord2 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Obtaining the filepath to the stored LINE models
model_path_ord1 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord1.pt")
model_path_ord2 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord2.pt")

# Loading the LINE models
line_ord1.load_state_dict(torch.load(model_path_ord1))
line_ord2.load_state_dict(torch.load(model_path_ord2))

line_ord1.eval()
line_ord2.eval()

# Extracting the node embeddings and context-node embeddings
embeddings_ord1 = line_ord1.node_embeddings.weight.data
embeddings_ord2 = line_ord2.contextnode_embeddings.weight.data

# Concatenating the embeddings
final_emb = torch.cat([embeddings_ord1,
                       embeddings_ord2], dim=1)

final_emb_np = final_emb.cpu().numpy()

# Instantiate a KMeans model
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init="auto", random_state=100)

# Fitting a KMeans model and getting the predicted labels (clusters) for each node
clusters = kmeans.fit_predict(final_emb_np)

# Creating a model to reduce the data to 2D using PCA (for plotting)
pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

# Obtaining the 2D node embeddings
pca_embeddings = pca_model.fit_transform(final_emb_np)

# Reading in the list of actual cluster labels (to compare the actual and predicted clusters)
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
node_blocks = sbm_data["Block/Cluster"]

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "..", "LINE_pca.png")

# Plot PCA of node embeddings and colour by actual cluster label
plt.scatter(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], c=node_blocks)
plt.savefig(output_path)

## Comparing KMeans classification with actual clusters

# Computing the normalized mutual info score (0 = no correlation, 1 = perfect correlation)
nmi_score = normalized_mutual_info_score(node_blocks, clusters)
print(f"NMI Score: {nmi_score}")