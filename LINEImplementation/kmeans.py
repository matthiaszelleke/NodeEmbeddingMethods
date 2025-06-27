import os
import torch
from config import args
from utils.utils import VoseAlias
from utils.line import LINE
from train import args
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle

MODEL_FILENAME = "LINE_model.pt"
NUM_CLUSTERS = 3
NUM_NODES = 1000
ANNOTATION_OFFSET = 0.04

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

line_ord1 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)
line_ord2 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path_ord1 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord1.pt")
model_path_ord2 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord2.pt")
line_ord1.load_state_dict(torch.load(model_path_ord1))
line_ord2.load_state_dict(torch.load(model_path_ord2))

line_ord1.eval()
line_ord2.eval()

embeddings_ord1 = line_ord1.node_embeddings.weight.data
embeddings_ord2 = line_ord2.contextnode_embeddings.weight.data

final_emb = torch.cat([embeddings_ord1,
                       embeddings_ord2], dim=1)

final_emb_np = final_emb.cpu().numpy()

# Instantiate a KMeans model
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=100)

# Fitting a KMeans model and getting the predicted labels (clusters) for each node
clusters = kmeans.fit_predict(final_emb_np)

# Creating a model to reduce the data to 2D using PCA (for plotting)
pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

# Obtaining the 2D node embeddings
pca_embeddings = pca_model.fit_transform(final_emb_np)

# Plot PCA of node embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "..", "..", "LINE_pca.png")

plt.scatter(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1])
plt.savefig(output_path)

## Comparing KMeans classification with actual clusters

# Reading in the list of actual cluster labels (to compare the actual and predicted clusters)
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
node_blocks = sbm_data["Block/Cluster"]

# Computing the normalized mutual info score (0 = no correlation, 1 = perfect correlation)
nmi_score = normalized_mutual_info_score(node_blocks, clusters)
print(f"NMI Score: {nmi_score}")