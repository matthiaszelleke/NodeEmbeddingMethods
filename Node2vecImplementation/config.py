import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-dim", "--dimensions", type=int, default=32)
parser.add_argument("-walk_length", "--walk_length", type=int, default=60)
parser.add_argument("-num_walks", "--num_walks", type=int, default=10)
parser.add_argument("-p", "--step_back_parameter", type=float, default=1.0)
parser.add_argument("-q", "--stay_local_parameter", type=float, default=1.0)
parser.add_argument("-window_size", "--window_size", type=int, default=10)
parser.add_argument("-emb_fname", "--embeddings_filename", type=str, default="spectral_clustering_node_embeddings.csv")

args, _ = parser.parse_known_args()