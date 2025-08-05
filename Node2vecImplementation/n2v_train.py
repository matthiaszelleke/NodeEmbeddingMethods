from .config import args
import networkx as nx
from node2vec import Node2Vec
import numpy as np

def get_embeddings_node2vec(network, dimensions=32, walk_length=60, num_walks=10, p=1, q=1, 
                            window_size=10, embeddings_filename="n2v_node_embeddings.csv"):

    ## Generating node embeddings for the SBM

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(
        network,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=4
    )

    # Embed nodes
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)

    # Save embeddings for later use
    model.wv.save_word2vec_format("Node2vecImplementation/"+embeddings_filename, write_header=False)

    node_ids = model.wv.index_to_key  # list of all node IDs as strings
    embeddings = np.array([model.wv[node_id] for node_id in node_ids])
    
    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network, nodetype=int)
    get_embeddings_node2vec(network, args.dimensions, args.walk_length, args.num_walks, args.p,
                            args.q, args.window_size, args.emb_fname)