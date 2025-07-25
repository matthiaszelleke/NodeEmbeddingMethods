from config import args
from grarep import GraRep
from grarep_utils import read_graph_edgelist

# Returns the graph's adjacency matrix
adj_matrix = read_graph_edgelist(args.graph_path)

grarep = GraRep(adj_matrix, args)
grarep.learn_embeddings()
grarep.save_embeddings()