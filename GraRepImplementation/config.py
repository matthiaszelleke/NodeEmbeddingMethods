import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--graph_path", type=str, default="sbm_graph.edgelist")
parser.add_argument("-embfile", "--embeddings_file", type=str, default="grarep_node_embeddings.csv")

# Hyperparameters
parser.add_argument("-order", "--order", type=int, default=6)
parser.add_argument("-dim", "--dimension", type=int, default=16)

args = parser.parse_args()