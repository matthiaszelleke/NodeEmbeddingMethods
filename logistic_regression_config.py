import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-emb_fname", "--embeddings_filename", type=str)
parser.add_argument("-model_and_labels_fname", "--model_and_labels_filename", type=str)
parser.add_argument("-metrics_fname", "--metrics_filename", type=str)

args, _ = parser.parse_known_args()