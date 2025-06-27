import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--graph_path", type=str)
parser.add_argument("-embfile", "--embeddings_file", type=str, default="GraRep_embeddings.csv")

# Hyperparameters
parser.add_argument("-order", "--order", type=int, default=6)
parser.add_argument("-dim", "--dimension", type=int, default=16)
parser.add_argument("-iters", "--iterations", type=int, default=20)

args = parser.parse_args()