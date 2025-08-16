import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-K", "--depth", type=int, help='Number of layers (K)', default=2)
parser.add_argument("-dim", "--embed_dim", type=int, default=32)
parser.add_argument("-patience", "--patience", type=int, default=2)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-weight_decay", "--weight_decay", type=float, default=0.0001)
parser.add_argument("-num_epochs", "--num_epochs", type=int, default=10)
parser.add_argument("-model_save", "--model_save_path", type=str, default="GraphSageImplementation/GraphSage_model.pt")
parser.add_argument("-emb_fname", "--embedding_filename", type=str, default="graphsage_node_embeddings.csv")

args, _ = parser.parse_known_args()