import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-network", "--network", type=str, default="sbm_graph.edgelist")
parser.add_argument("-K", "--depth", type=int, help='Number of layers (K)', default=2)
parser.add_argument("-neigh_sizes", "--neighbourhood-sizes", type=int, nargs='+',
    help='List of neighbourhood sample sizes per layer, e.g. --neighbourhood-sizes 10 5',
    default=[10, 5])

parser.add_argument("-dim", "--embed_dim", type=int, default=32)
parser.add_argument("-num_learn_feat", "--num_learn_feat", type=int, default=-1, 
                    help="The number of node features to be learned by GraphSage, " \
                    "only use if network doesn't have node features")

parser.add_argument("-num_walks", "--num_walks", type=int, default=10)
parser.add_argument("-walk_length", "--walk_length", type=int, default=80)
parser.add_argument("-w", "--window_size", type=int, default=5)
parser.add_argument("-Q", "--neg_sample_size", type=int, default=5)

parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-num_epochs", "--num_epochs", type=int, default=10)
parser.add_argument("-model_save", "--model_save_path", type=str, default="GraphSageImplementation/GraphSage_model.pt")
parser.add_argument("-emb_fname", "--embedding_filename", type=str, default="graphsage_node_embeddings.csv")

args, _ = parser.parse_known_args()