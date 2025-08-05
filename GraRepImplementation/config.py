import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-emb_fname", "--embeddings_filename", type=str, default="grarep_node_embeddings.csv")
parser.add_argument("-order", "--order", type=int, default=6)
parser.add_argument("-dim", "--dimension", type=int, default=16)

args, _ = parser.parse_known_args()