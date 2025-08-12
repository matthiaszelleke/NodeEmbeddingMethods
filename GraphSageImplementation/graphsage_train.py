from .config import args
import networkx as nx
import numpy as np
import random
import torch
import torch.optim as optim
from tqdm import trange
from .utils.graphsage import GraphSage
from .utils.utils import loss_function

def get_batch_size(num_nodes):
    if num_nodes < 500:
        return num_nodes       # small graphs — large batches
    elif num_nodes < 5_000:
        return 256       # small-medium graphs
    elif num_nodes < 50_000:
        return 128       # medium graphs
    elif num_nodes < 500_000:
        return 64        # large graphs
    else:
        return 32        # very large graphs

def setup(network, K, neigh_sizes, embed_dim=32, num_learnable_features=None,
          lr=0.001, num_walks=10, walk_length=80, window_size=5, neg_sample_size=5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_sage = GraphSage(network, neigh_sizes, K, embed_dim, num_learnable_features,
                           num_walks, walk_length, window_size, 
                           neg_sample_size)
    opt = optim.Adam(graph_sage.parameters(), lr=lr)

    return graph_sage, opt, device

def train(graph_sage, opt, num_epochs, device):

    print(f"\nTraining on {device}...\n")
    batch_size = get_batch_size(graph_sage.num_nodes)
    num_batches = (graph_sage.num_nodes + batch_size - 1) // batch_size

    all_nodes = list(range(1, graph_sage.num_nodes+1))
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")

        random.shuffle(all_nodes)
        for batch_num in trange(num_batches):
            batch_nodes = all_nodes[batch_num*batch_size : (batch_num+1)*batch_size]

            graph_sage.zero_grad()
            embeddings = graph_sage(batch_nodes, device)
            loss = loss_function(embeddings, graph_sage.network, batch_size, graph_sage.nodealiassampler, 
                                 graph_sage.num_walks, graph_sage.walk_length, graph_sage.window_size, 
                                 graph_sage.neg_sample_size)
            loss.backward()
            opt.step()

    return graph_sage

def save(model, embeddings, model_save_path, embeddings_filename):
    
    print(f"\nDone training, saving model at GraphSageImplementation/{model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    embeddings = embeddings.weight.data.cpu().numpy()

    print(f"Saving embeddings at GraphSageImplementation/{embeddings_filename}")
    np.savetxt("GraphSageImplementation/"+embeddings_filename, embeddings, delimiter=",", fmt='%.6f')

    return embeddings

def get_embeddings_graphsage(network, num_epochs, K, neigh_sizes, embed_dim=32, num_learnable_features=None, 
                             lr=0.001, num_walks=10, walk_length=80, window_size=5, neg_sample_size=5,
                             embeddings_filename="graphsage_node_embeddings.csv", 
                             model_save_path="GraphSageImplementation/GraphSage_model.pt"):
    
    graph_sage, opt, device = setup(network, K, neigh_sizes, embed_dim, num_learnable_features,
                                    lr, num_walks, walk_length, window_size, neg_sample_size)
    
    graph_sage = train(graph_sage, opt, num_epochs, device)
    
    embeddings = save(graph_sage, graph_sage.node_embeddings, model_save_path, embeddings_filename)

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network)
    get_embeddings_graphsage(network, args.num_epochs, args.K, args.neigh_sizes, args.dim, 
                             args.num_learn_feat, args.lr, args.num_walks, args.walk_length, 
                             args.window_size, args.Q, args.emb_fname, args.model_save_path)