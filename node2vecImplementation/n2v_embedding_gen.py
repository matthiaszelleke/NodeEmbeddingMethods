# This script is adapted from example usage code in:
# https://github.com/eliorc/node2vec
# by Elior Cohen, MIT License


from node2vec import Node2Vec
import pickle

EMBEDDING_FILENAME = "n2v_node_embeddings.txt"

## Generating node embeddings for the SBM

# Reading in the graph object
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
G = sbm_data["Graph"]

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(
    G,
    dimensions=128,
    workers=4
)

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME, write_header=False)
