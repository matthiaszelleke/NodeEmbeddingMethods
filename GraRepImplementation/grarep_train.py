from .config import args
from .grarep import GraRep
import networkx as nx


def get_embeddings_grarep(network, order=6, dimensions=21, embeddings_filename="grarep_node_embeddings.csv"):

    adj_matrix = nx.adjacency_matrix(network).toarray()

    grarep = GraRep(adj_matrix, order, dimensions, embeddings_filename)
    embeddings = grarep.learn_embeddings()
    grarep.save_embeddings()

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network)
    get_embeddings_grarep(network, args.order, args.dim, args.emb_fnmae)