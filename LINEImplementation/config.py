# Adapted from: https://github.com/dmpierre/LINE
# Copyright (c) 2025 Pierre Daix-Moreux
# Licensed under the MIT License (see LICENSE or LICENSE-pierre in this repository)
# Implements the LINE algorithm from:
# Tang et al. (2015). "LINE: Large-scale Information Network Embedding"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-save", "--save_path", type=str, default="LINEImplementation/LINE_model.pt")
parser.add_argument("-loss_data", "--loss_data_path", type=str, default="LINEImplementation/loss_data.pkl")
parser.add_argument("-neg", "--neg_sample_size", type=int, default=5)
parser.add_argument("-dim", "--dimension", type=int, default=64)
parser.add_argument("-batch_size", "--batch_size", type=int, default=64)
parser.add_argument("-epochs", "--epochs", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)  # As starting value in paper
parser.add_argument("-neg_pow", "--negative_power", type=float, default=0.75)
parser.add_argument("-emb_fname", "--embedding_filename", type=str, default="line_node_embeddings.csv")

args, _ = parser.parse_known_args()