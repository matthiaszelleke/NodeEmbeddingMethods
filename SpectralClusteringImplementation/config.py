import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-emb_fname", "--embeddings_filename", type=str, default="spectral_clustering_node_embeddings.csv")

args, _ = parser.parse_known_args()